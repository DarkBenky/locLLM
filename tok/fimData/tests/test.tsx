export function add(a: number, b: number): number {
    return a + b;
}

export const Button = (props: { label: string }) => {
    return <button>{props.label}</button>;
};

export class App extends Component {
    render() {
        return <div />;
    }
}
